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
        tagValue2 = new TagValue("tag1", "value1");
        tagValue3 = new TagValue("tag2", "value2");
    }

    @Test
    void testEquals_SameInstance() {
        assertTrue(tagValue1.equals(tagValue1));
    }

    @Test
    void testEquals_NullObject() {
        assertFalse(tagValue1.equals(null));
    }

    @Test
    void testEquals_DifferentClass() {
        assertFalse(tagValue1.equals(new Object()));
    }

    @Test
    void testEquals_EqualObjects() {
        try (MockedStatic<Util> mockedUtil = Mockito.mockStatic(Util.class)) {
            mockedUtil.when(() -> Util.StringCompare("tag1", "tag1")).thenReturn(0);
            mockedUtil.when(() -> Util.StringCompare("value1", "value1")).thenReturn(0);
            assertTrue(tagValue1.equals(tagValue2));
        }
    }

    @Test
    void testEquals_DifferentTags() {
        try (MockedStatic<Util> mockedUtil = Mockito.mockStatic(Util.class)) {
            mockedUtil.when(() -> Util.StringCompare("tag1", "tag2")).thenReturn(1);
            mockedUtil.when(() -> Util.StringCompare("value1", "value2")).thenReturn(0);
            assertFalse(tagValue1.equals(tagValue3));
        }
    }

    @Test
    void testEquals_DifferentValues() {
        try (MockedStatic<Util> mockedUtil = Mockito.mockStatic(Util.class)) {
            mockedUtil.when(() -> Util.StringCompare("tag1", "tag1")).thenReturn(0);
            mockedUtil.when(() -> Util.StringCompare("value1", "value2")).thenReturn(1);
            assertFalse(tagValue1.equals(tagValue3));
        }
    }
}
