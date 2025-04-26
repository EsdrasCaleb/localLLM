package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class TagValue_equals_0_1_Test {

    @InjectMocks
    private TagValue tagValue;

    @Mock
    private Util util;

    @Test
    public void testEquals_sameObject_returnsTrue() {
        // Arrange
        TagValue other = new TagValue("tag", "value");
        // Act
        boolean result = tagValue.equals(other);
        // Assert
        assertTrue(result);
    }

    @Test
    public void testEquals_null_returnsFalse() {
        // Arrange
        TagValue other = null;
        // Act
        boolean result = tagValue.equals(other);
        // Assert
        assertFalse(result);
    }

    @Test
    public void testEquals_differentObject_returnsFalse() {
        // Arrange
        TagValue other = new TagValue("otherTag", "otherValue");
        // Act
        boolean result = tagValue.equals(other);
        // Assert
        assertFalse(result);
    }

    @Test
    public void testEquals_differentTag_returnsFalse() {
        // Arrange
        TagValue other = new TagValue("otherTag", "value");
        // Act
        boolean result = tagValue.equals(other);
        // Assert
        assertFalse(result);
    }

    @Test
    public void testEquals_differentValue_returnsFalse() {
        // Arrange
        TagValue other = new TagValue("tag", "otherValue");
        // Act
        boolean result = tagValue.equals(other);
        // Assert
        assertFalse(result);
    }

    @Test
    public void testEquals_sameTagDifferentValue_returnsTrue() {
        // Arrange
        TagValue other = new TagValue("tag", "otherValue");
        // Act
        boolean result = tagValue.equals(other);
        // Assert
        assertTrue(result);
    }
}
