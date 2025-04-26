package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class TagValue_equals_0_3_Test {

    @Test
    public void testEquals() {
        // Arrange
        TagValue obj1 = new TagValue("tag1", "value1");
        TagValue obj2 = new TagValue("tag1", "value1");
        TagValue obj3 = new TagValue("tag2", "value2");
        // Act
        boolean result = obj1.equals(obj2);
        // Assert
        assertEquals(true, result);
    }

    @Test
    public void testNotEquals() {
        // Arrange
        TagValue obj1 = new TagValue("tag1", "value1");
        TagValue obj2 = new TagValue("tag2", "value2");
        TagValue obj3 = new TagValue("tag1", "value2");
        // Act
        boolean result = obj1.equals(obj2);
        // Assert
        assertEquals(false, result);
    }
}
