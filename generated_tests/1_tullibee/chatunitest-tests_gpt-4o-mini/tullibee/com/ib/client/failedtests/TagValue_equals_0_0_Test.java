package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class TagValue_equals_0_0_Test {

    private TagValue tagValue1;

    private TagValue tagValue2;

    private TagValue tagValue3;

    @BeforeEach
    public void setUp() {
        tagValue1 = new TagValue("tag1", "value1");
        tagValue2 = new TagValue("tag1", "value1");
        tagValue3 = new TagValue("tag2", "value2");
    }

    @Test
    public void testEquals_SameReference() {
        assertTrue(tagValue1.equals(tagValue1), "Should be equal to itself");
    }

    @Test
    public void testEquals_Null() {
        assertFalse(tagValue1.equals(null), "Should not be equal to null");
    }

    @Test
    public void testEquals_DifferentClass() {
        assertFalse(tagValue1.equals(new Object()), "Should not be equal to an object of different class");
    }

    @Test
    public void testEquals_EqualObjects() {
        assertTrue(tagValue1.equals(tagValue2), "Should be equal to another object with same values");
    }

    @Test
    public void testEquals_DifferentTag() {
        assertFalse(tagValue1.equals(tagValue3), "Should not be equal to an object with different tag");
    }

    @Test
    public void testEquals_DifferentValue() {
        TagValue tagValue4 = new TagValue("tag1", "value2");
        assertFalse(tagValue1.equals(tagValue4), "Should not be equal to an object with different value");
    }
}
