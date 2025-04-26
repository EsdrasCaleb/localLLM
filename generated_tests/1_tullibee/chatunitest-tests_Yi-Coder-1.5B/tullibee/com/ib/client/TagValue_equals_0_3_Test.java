package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
public class TagValue_equals_0_3_Test {

    @Test
    public void testEquals() {
        TagValue l_tag = new TagValue("tag1", "value1");
        TagValue l_tag2 = new TagValue("tag1", "value1");
        TagValue l_tag3 = new TagValue("tag1", "value2");
        TagValue l_tag4 = new TagValue("tag2", "value1");
        TagValue l_tag5 = new TagValue("tag3", "value1");
        assertTrue(l_tag.equals(l_tag2));
        assertTrue(l_tag.equals(l_tag3));
        assertFalse(l_tag.equals(l_tag4));
        assertFalse(l_tag.equals(l_tag5));
    }
}
