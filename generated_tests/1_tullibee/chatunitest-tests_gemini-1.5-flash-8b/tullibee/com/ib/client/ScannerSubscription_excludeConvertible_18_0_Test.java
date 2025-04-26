package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_excludeConvertible_18_0_Test {

    @Test
    void testExcludeConvertible_validInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        String expectedValue = "true";
        try {
            Field field = ScannerSubscription.class.getDeclaredField("m_excludeConvertible");
            field.setAccessible(true);
            field.set(subscription, expectedValue);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing private field: " + e.getMessage());
        }
        String actualValue = subscription.excludeConvertible();
        assertEquals(expectedValue, actualValue);
    }

    @Test
    void testExcludeConvertible_nullInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        String expectedValue = null;
        try {
            Field field = ScannerSubscription.class.getDeclaredField("m_excludeConvertible");
            field.setAccessible(true);
            field.set(subscription, expectedValue);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing private field: " + e.getMessage());
        }
        String actualValue = subscription.excludeConvertible();
        assertEquals(expectedValue, actualValue);
    }

    @Test
    void testExcludeConvertible_emptyInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        String expectedValue = "";
        try {
            Field field = ScannerSubscription.class.getDeclaredField("m_excludeConvertible");
            field.setAccessible(true);
            field.set(subscription, expectedValue);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing private field: " + e.getMessage());
        }
        String actualValue = subscription.excludeConvertible();
        assertEquals(expectedValue, actualValue);
    }
}
