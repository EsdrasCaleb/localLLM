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
        // Create two TagValue objects with the same tag and value
        TagValue tagValue1 = new TagValue("tag1", "value1");
        TagValue tagValue2 = new TagValue("tag1", "value1");
        // Check if they are equal
        assertTrue(tagValue1.equals(tagValue2));
        // Create two TagValue objects with different tags
        TagValue tagValue3 = new TagValue("tag1", "value2");
        TagValue tagValue4 = new TagValue("tag2", "value2");
        // Check if they are not equal
        assertFalse(tagValue3.equals(tagValue4));
    }
}
