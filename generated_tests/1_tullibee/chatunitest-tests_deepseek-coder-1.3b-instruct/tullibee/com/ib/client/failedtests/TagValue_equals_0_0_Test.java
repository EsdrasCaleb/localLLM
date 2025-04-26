package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class TagValue_equals_0_0_Test {

    @Test
    void testEquals() {
        // Arrange
        String tag = "tag1";
        String value = "value1";
        TagValue tagValue1 = new TagValue(tag, value);
        TagValue tagValue2 = new TagValue(tag, value);
        // Act
        boolean result = tagValue1.equals(tagValue2);
        // Assert
        assertTrue(result);
    }

    @Test
    void testNotEquals() {
        // Arrange
        String tag = "tag1";
        String value = "value1";
        TagValue tagValue1 = new TagValue(tag, value);
        TagValue tagValue2 = new TagValue("tag2", "value2");
        // Act
        boolean result = tagValue1.equals(tagValue2);
        // Assert
        assertFalse(result);
    }

    @Test
    void testNull() {
        // Arrange
        String tag = "tag1";
        String value = "value1";
        TagValue tagValue1 = new TagValue(tag, value);
        // Act
        boolean result = tagValue1.equals(null);
        // Assert
        assertFalse(result);
    }

    @Test
    void testNotObject() {
        // Arrange
        String tag = "tag1";
        String value = "value1";
        TagValue tagValue1 = new TagValue(tag, value);
        // Act
        boolean result = tagValue1.equals("object");
        // Assert
        assertFalse(result);
    }

    @Test
    void testSameObject() {
        // Arrange
        String tag = "tag1";
        String value = "value1";
        TagValue tagValue1 = new TagValue(tag, value);
        // Act
        boolean result = tagValue1.equals(tagValue1);
        // Assert
        assertTrue(result);
    }
}
