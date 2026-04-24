/**
 * Generates a unique ID using crypto.randomUUID when available,
 * falls back to a timestamp + random string.
 */
export function generateId(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return 'id-' + Date.now() + '-' + Math.random().toString(36).substring(2, 11);
}

/**
 * Simple language detection based on character ranges.
 * Returns 'chinese', 'vietnamese', or 'english'.
 */
export function detectLanguage(text: string): 'chinese' | 'vietnamese' | 'english' {
  const trimmed = text.trim();
  if (!trimmed) return 'english';

  // Check for Chinese characters (CJK Unified Ideographs)
  const chineseRegex = /[\u4e00-\u9fff\u3400-\u4dbf]/;
  if (chineseRegex.test(trimmed)) {
    return 'chinese';
  }

  // Check for Vietnamese-specific characters (diacritics: ă, â, đ, ê, ô, ơ, ư)
  const vietnameseRegex = /[áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡựúùủũụýỳỷỹỵđ]/i;
  if (vietnameseRegex.test(trimmed)) {
    return 'vietnamese';
  }

  return 'english';
}

/**
 * Formats a Date object into a readable time string.
 */
export function formatTimestamp(date: Date): string {
  const dateObj = date instanceof Date ? date : new Date(date);
  return dateObj.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
