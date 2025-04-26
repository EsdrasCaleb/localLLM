package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class TagValue_equals_0_1_Test {

    @Test
    public void test_equals_with_same_tag_and_value() {
        TagValue l_tagValue1 = new TagValue("tag1", "value1");
        TagValue l_tagValue2 = new TagValue("tag1", "value1");
        assertEquals(l_tagValue1, l_tagValue2);
    }

    @Test
    public void test_equals_with_different_tag_and_value() {
        TagValue l_tagValue1 = new TagValue("tag1", "value1");
        TagValue l_tagValue2 = new TagValue("tag2", "value2");
        assertNotEquals(l_tagValue1, l_tagValue2);
    }

    @Test
    public void test_equals_with_null_object() {
        TagValue l_tagValue = new TagValue("tag1", "value1");
        assertNotEquals(l_tagValue, null);
    }
}
