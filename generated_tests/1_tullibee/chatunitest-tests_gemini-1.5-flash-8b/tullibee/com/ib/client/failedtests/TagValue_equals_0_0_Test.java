package com.ib.client;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class TagValue_equals_0_0_Test {

    @Test
    void testEquals_sameObject() {
        TagValue obj = new TagValue("tag1", "value1");
        assertTrue(obj.equals(obj));
    }

    @Test
    void testEquals_nullObject() {
        TagValue obj = new TagValue("tag1", "value1");
        assertFalse(obj.equals(null));
    }

    @Test
    void testEquals_differentClass() {
        TagValue obj = new TagValue("tag1", "value1");
        assertFalse(obj.equals(new Object()));
    }

    @Test
    void testEquals_differentTags() {
        TagValue obj1 = new TagValue("tag1", "value1");
        TagValue obj2 = new TagValue("tag2", "value1");
        assertFalse(obj1.equals(obj2));
    }

    @Test
    void testEquals_differentValues() {
        TagValue obj1 = new TagValue("tag1", "value1");
        TagValue obj2 = new TagValue("tag1", "value2");
        assertFalse(obj1.equals(obj2));
    }

    @Test
    void testEquals_sameTagsAndValues() {
        TagValue obj1 = new TagValue("tag1", "value1");
        TagValue obj2 = new TagValue("tag1", "value1");
        assertTrue(obj1.equals(obj2));
    }

    @Test
    void testEquals_nullTag() {
        TagValue obj1 = new TagValue(null, "value1");
        TagValue obj2 = new TagValue(null, "value1");
        assertTrue(obj1.equals(obj2));
    }

    @Test
    void testEquals_nullValue() {
        TagValue obj1 = new TagValue("tag1", null);
        TagValue obj2 = new TagValue("tag1", null);
        assertTrue(obj1.equals(obj2));
    }

    @Test
    void testEquals_nullTagAndValue() {
        TagValue obj1 = new TagValue(null, null);
        TagValue obj2 = new TagValue(null, null);
        assertTrue(obj1.equals(obj2));
    }
}
