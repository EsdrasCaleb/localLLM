package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class TagValue_equals_0_0_Test {

    @Mock
    private Util util;

    @InjectMocks
    private TagValue tagValue;

    @Test
    public void testEqualsNull() {
        // Arrange
        TagValue other = null;
        // Act and Assert
        boolean result = tagValue.equals(other);
        assertFalse(result);
    }

    @Test
    public void testEqualsDifferentTags() {
        // Arrange
        TagValue other = new TagValue("test", "value");
        when(util.StringCompare(tagValue.m_tag, other.m_tag)).thenReturn(1);
        // Act and Assert
        boolean result = tagValue.equals(other);
        assertFalse(result);
    }

    @Test
    public void testEqualsDifferentValues() {
        // Arrange
        TagValue other = new TagValue("test", "value");
        when(util.StringCompare(tagValue.m_value, other.m_value)).thenReturn(1);
        // Act and Assert
        boolean result = tagValue.equals(other);
        assertFalse(result);
    }

    @Test
    public void testEqualsSameTagsAndValues() {
        // Arrange
        TagValue other = new TagValue("test", "value");
        when(util.StringCompare(tagValue.m_tag, other.m_tag)).thenReturn(0);
        when(util.StringCompare(tagValue.m_value, other.m_value)).thenReturn(0);
        // Act and Assert
        boolean result = tagValue.equals(other);
        assertTrue(result);
    }

    @Test
    public void testEqualsSameObject() {
        // Arrange
        TagValue other = tagValue;
        // Act and Assert
        boolean result = tagValue.equals(other);
        assertTrue(result);
    }
}
