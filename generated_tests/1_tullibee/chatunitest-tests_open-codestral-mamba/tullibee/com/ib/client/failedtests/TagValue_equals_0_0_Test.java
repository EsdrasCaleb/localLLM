package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class TagValue_equals_0_0_Test {

    private TagValue tagValue1;

    private TagValue tagValue2;

    private TagValue tagValue3;

    @BeforeEach
    void setUp() {
        tagValue1 = new TagValue("tag1", "value1");
        tagValue2 = new TagValue("tag2", "value2");
        tagValue3 = new TagValue("tag1", "value1");
    }

    @Test
    void testEquals_SameObject_ReturnsTrue() {
        assertTrue(tagValue1.equals(tagValue1));
    }

    @Test
    void testEquals_Null_ReturnsFalse() {
        assertFalse(tagValue1.equals(null));
    }

    @Test
    void testEquals_DifferentObject_ReturnsFalse() {
        assertFalse(tagValue1.equals(new Object()));
    }

    @Test
    void testEquals_DifferentTag_ReturnsFalse() {
        assertFalse(tagValue1.equals(tagValue2));
    }

    @Test
    void testEquals_DifferentValue_ReturnsFalse() {
        assertFalse(tagValue1.equals(tagValue3));
    }

    @Test
    void testEquals_SameTagAndValue_ReturnsTrue() {
        assertTrue(tagValue1.equals(tagValue3));
    }
}
