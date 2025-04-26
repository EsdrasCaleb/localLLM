package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class TagValue_equals_0_0_Test {

    @Test
    public void testEquals_ObjectIsSameInstance_ReturnsTrue() {
        TagValue obj1 = new TagValue("tag1", "value1");
        TagValue obj2 = obj1;
        assertTrue(obj1.equals(obj2));
    }

    @Test
    public void testEquals_NullObject_ReturnsFalse() {
        TagValue obj1 = new TagValue("tag1", "value1");
        TagValue obj2 = null;
        assertFalse(obj1.equals(obj2));
    }

    @Test
    public void testEquals_DifferentInstances_ReturnsFalse() {
        TagValue obj1 = new TagValue("tag1", "value1");
        TagValue obj2 = new TagValue("tag2", "value2");
        assertFalse(obj1.equals(obj2));
    }

    @Test
    public void testEquals_MismatchedTags_ReturnsFalse() {
        TagValue obj1 = new TagValue("tag1", "value1");
        TagValue obj2 = new TagValue("tag2", "value1");
        assertFalse(obj1.equals(obj2));
    }

    @Test
    public void testEquals_MismatchedValues_ReturnsFalse() {
        TagValue obj1 = new TagValue("tag1", "value1");
        TagValue obj2 = new TagValue("tag1", "value2");
        assertFalse(obj1.equals(obj2));
    }

    @Test
    public void testEquals_MatchedTagsAndValues_ReturnsTrue() {
        TagValue obj1 = new TagValue("tag1", "value1");
        TagValue obj2 = new TagValue("tag1", "value1");
        assertTrue(obj1.equals(obj2));
    }

    @Test
    public void testEquals_MatchedTagsAndValuesWithDifferentCase_ReturnsTrue() {
        TagValue obj1 = new TagValue("tag1", "value1");
        TagValue obj2 = new TagValue("TAG1", "VALUE1");
        assertTrue(obj1.equals(obj2));
    }

    @Test
    public void testEquals_MatchedTagsAndValuesWithDifferentLength_ReturnsTrue() {
        TagValue obj1 = new TagValue("tag1", "value1");
        TagValue obj2 = new TagValue("tag1", "value1");
        assertTrue(obj1.equals(obj2));
    }

    @Test
    public void testEquals_MismatchedTagsAndValues_ReturnsFalse() {
        TagValue obj1 = new TagValue("tag1", "value1");
        TagValue obj2 = new TagValue("tag2", "value2");
        assertFalse(obj1.equals(obj2));
    }

    @Test
    public void testEquals_MismatchedTagsAndValuesWithDifferentCase_ReturnsFalse() {
        TagValue obj1 = new TagValue("tag1", "value1");
        TagValue obj2 = new TagValue("TAG2", "VALUE1");
        assertFalse(obj1.equals(obj2));
    }

    @Test
    public void testEquals_MismatchedTagsAndValuesWithDifferentLength_ReturnsFalse() {
        TagValue obj1 = new TagValue("tag1", "value1");
        TagValue obj2 = new TagValue("tag1", "value2");
        assertFalse(obj1.equals(obj2));
    }
}
