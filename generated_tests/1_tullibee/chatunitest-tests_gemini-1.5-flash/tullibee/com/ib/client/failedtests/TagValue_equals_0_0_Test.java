package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class TagValue_equals_0_0_Test {

    // Mock Util class since the real implementation is not provided
    static class Util {

        public static int StringCompare(String s1, String s2) {
            if (s1 == null && s2 == null)
                return 0;
            if (s1 == null)
                return -1;
            if (s2 == null)
                return 1;
            return s1.compareTo(s2);
        }
    }

    @Test
    void testEquals_SameObject() {
        TagValue tv = new TagValue("test", "value");
        assertTrue(tv.equals(tv));
    }

    @Test
    void testEquals_NullObject() {
        TagValue tv = new TagValue("test", "value");
        assertFalse(tv.equals(null));
    }

    @Test
    void testEquals_DifferentObject() {
        TagValue tv1 = new TagValue("test1", "value1");
        TagValue tv2 = new TagValue("test2", "value2");
        assertFalse(tv1.equals(tv2));
    }

    @Test
    void testEquals_EqualObject() {
        TagValue tv1 = new TagValue("test", "value");
        TagValue tv2 = new TagValue("test", "value");
        assertTrue(tv1.equals(tv2));
    }

    @Test
    void testEquals_NullTag() throws NoSuchFieldException, IllegalAccessException {
        TagValue tv1 = new TagValue("test", "value");
        TagValue tv2 = new TagValue(null, "value");
        Field tagField = TagValue.class.getDeclaredField("m_tag");
        tagField.setAccessible(true);
        tagField.set(tv1, null);
        assertFalse(tv1.equals(tv2));
    }

    @Test
    void testEquals_NullValue() throws NoSuchFieldException, IllegalAccessException {
        TagValue tv1 = new TagValue("test", "value");
        TagValue tv2 = new TagValue("test", null);
        Field valueField = TagValue.class.getDeclaredField("m_value");
        valueField.setAccessible(true);
        valueField.set(tv1, null);
        assertFalse(tv1.equals(tv2));
    }

    @Test
    void testEquals_DifferentTag() {
        TagValue tv1 = new TagValue("test1", "value");
        TagValue tv2 = new TagValue("test2", "value");
        assertFalse(tv1.equals(tv2));
    }

    @Test
    void testEquals_DifferentValue() {
        TagValue tv1 = new TagValue("test", "value1");
        TagValue tv2 = new TagValue("test", "value2");
        assertFalse(tv1.equals(tv2));
    }

    @Test
    void testEquals_NullTagAndValue() throws NoSuchFieldException, IllegalAccessException {
        TagValue tv1 = new TagValue("test", "value");
        TagValue tv2 = new TagValue(null, null);
        Field tagField = TagValue.class.getDeclaredField("m_tag");
        tagField.setAccessible(true);
        tagField.set(tv1, null);
        Field valueField = TagValue.class.getDeclaredField("m_value");
        valueField.setAccessible(true);
        valueField.set(tv1, null);
        assertTrue(tv1.equals(tv2));
    }

    @Test
    void testEquals_DifferentClass() {
        TagValue tv1 = new TagValue("test", "value");
        assertFalse(tv1.equals(new Object()));
    }
}
