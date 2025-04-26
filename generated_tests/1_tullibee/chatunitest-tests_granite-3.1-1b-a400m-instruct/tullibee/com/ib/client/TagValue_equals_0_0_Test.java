package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class TagValue_equals_0_0_Test {

    @Test
    void testEquals() {
        // Test case 1: Two TagValue objects with the same tag and value
        TagValue tag1 = new TagValue("testTag", "testValue");
        TagValue tag2 = new TagValue("testTag", "testValue");
        assertTrue(tag1.equals(tag2));
        // Test case 2: Two TagValue objects with different tag and value
        TagValue tag3 = new TagValue("testTag2", "testValue2");
        TagValue tag4 = new TagValue("testTag1", "testValue1");
        assertFalse(tag1.equals(tag3));
        assertFalse(tag1.equals(tag4));
        // Test case 3: TagValue object is null
        TagValue tag5 = null;
        assertFalse(tag1.equals(tag5));
    }
}
