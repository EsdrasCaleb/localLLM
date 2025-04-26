package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class TagValue_equals_0_0_Test {

    @Test
    public void testEquals() {
        // Arrange
        TagValue tagValue1 = new TagValue("tag1", "value1");
        TagValue tagValue2 = new TagValue("tag2", "value2");
        // Act
        boolean result = tagValue1.equals(tagValue2);
        // Assert
        assertFalse(result);
    }

    @Test
    public void testEqualsSameObject() {
        // Arrange
        TagValue tagValue = new TagValue("tag", "value");
        // Act
        boolean result = tagValue.equals(tagValue);
        // Assert
        assertTrue(result);
    }

    @Test
    public void testEqualsNull() {
        // Arrange
        TagValue tagValue = new TagValue("tag", "value");
        // Act
        boolean result = tagValue.equals(null);
        // Assert
        assertFalse(result);
    }

    @Test
    public void testEqualsDifferentObject() {
        // Arrange
        TagValue tagValue1 = new TagValue("tag1", "value1");
        TagValue tagValue2 = new TagValue("tag2", "value2");
        // Act
        boolean result = tagValue1.equals(tagValue2);
        // Assert
        assertFalse(result);
    }

    @Test
    public void testEqualsDifferentTag() {
        // Arrange
        TagValue tagValue1 = new TagValue("tag1", "value1");
        TagValue tagValue2 = new TagValue("tag2", "value2");
        // Act
        boolean result = tagValue1.equals(tagValue2);
        // Assert
        assertFalse(result);
    }

    @Test
    public void testEqualsDifferentValue() {
        // Arrange
        TagValue tagValue1 = new TagValue("tag1", "value1");
        TagValue tagValue2 = new TagValue("tag1", "value2");
        // Act
        boolean result = tagValue1.equals(tagValue2);
        // Assert
        assertFalse(result);
    }
}
