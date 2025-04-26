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
        TagValue tagValue1 = new TagValue("tag1", "value1");
        TagValue tagValue2 = new TagValue("tag1", "value1");
        TagValue tagValue3 = new TagValue("tag2", "value2");
        TagValue tagValue4 = new TagValue("tag1", "value2");
        assertTrue(tagValue1.equals(tagValue2));
        assertFalse(tagValue1.equals(tagValue3));
        assertFalse(tagValue1.equals(tagValue4));
    }
}
