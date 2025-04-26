package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class TagValue_equals_0_0_Test {

    @Mock
    private Util mockUtil;

    private TagValue tagValue;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        tagValue = new TagValue("tag1", "value1");
    }

    @Test
    public void testEquals_SameInstance() {
        assertTrue(tagValue.equals(tagValue));
    }

    @Test
    public void testEquals_NullObject() {
        assertFalse(tagValue.equals(null));
    }

    @Test
    public void testEquals_DifferentClass() {
        assertFalse(tagValue.equals("someString"));
    }

    @Test
    public void testEquals_EqualTagValue() {
        TagValue otherTagValue = new TagValue("tag1", "value1");
        when(mockUtil.StringCompare("tag1", "tag1")).thenReturn(0);
        when(mockUtil.StringCompare("value1", "value1")).thenReturn(0);
        assertTrue(tagValue.equals(otherTagValue));
    }

    @Test
    public void testEquals_DifferentTag() {
        TagValue otherTagValue = new TagValue("tag2", "value1");
        when(mockUtil.StringCompare("tag1", "tag2")).thenReturn(1);
        assertFalse(tagValue.equals(otherTagValue));
    }

    @Test
    public void testEquals_DifferentValue() {
        TagValue otherTagValue = new TagValue("tag1", "value2");
        when(mockUtil.StringCompare("tag1", "tag1")).thenReturn(0);
        when(mockUtil.StringCompare("value1", "value2")).thenReturn(1);
        assertFalse(tagValue.equals(otherTagValue));
    }

    @Test
    public void testEquals_DifferentTagAndValue() {
        TagValue otherTagValue = new TagValue("tag2", "value2");
        when(mockUtil.StringCompare("tag1", "tag2")).thenReturn(1);
        when(mockUtil.StringCompare("value1", "value2")).thenReturn(1);
        assertFalse(tagValue.equals(otherTagValue));
    }
}

// Mock Util class
class Util {

    public static int StringCompare(String s1, String s2) {
        return s1.compareTo(s2);
    }
}
