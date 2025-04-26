package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_excludeConvertible_39_0_Test {

    @Test
    void testExcludeConvertibleNull() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        setExcludeConvertible(subscription, null);
        assertEquals(null, getExcludeConvertible(subscription));
    }

    @Test
    void testExcludeConvertibleEmptyString() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        setExcludeConvertible(subscription, "");
        assertEquals("", getExcludeConvertible(subscription));
    }

    @Test
    void testExcludeConvertibleNonEmptyString() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        String testString = "testString";
        setExcludeConvertible(subscription, testString);
        assertEquals(testString, getExcludeConvertible(subscription));
    }

    @Test
    void testExcludeConvertibleWhitespaceString() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        setExcludeConvertible(subscription, "   ");
        assertEquals("   ", getExcludeConvertible(subscription));
    }

    @Test
    void testM_excludeConvertibleFieldAccess() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        Field field = ScannerSubscription.class.getDeclaredField("m_excludeConvertible");
        field.setAccessible(true);
        String initialValue = (String) field.get(subscription);
        assertNull(initialValue);
        subscription.excludeConvertible("Test Value");
        assertEquals("Test Value", field.get(subscription));
    }

    private void setExcludeConvertible(ScannerSubscription subscription, String value) throws NoSuchFieldException, IllegalAccessException {
        Field field = ScannerSubscription.class.getDeclaredField("m_excludeConvertible");
        field.setAccessible(true);
        field.set(subscription, value);
    }

    private String getExcludeConvertible(ScannerSubscription subscription) throws NoSuchFieldException, IllegalAccessException {
        Field field = ScannerSubscription.class.getDeclaredField("m_excludeConvertible");
        field.setAccessible(true);
        return (String) field.get(subscription);
    }
}
