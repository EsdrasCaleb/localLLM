package com.ib.client;

import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_maturityDateAbove_14_0_Test {

    @Test
    void testMaturityDateAbove_validInput() throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {
        ScannerSubscription subscription = new ScannerSubscription();
        String expectedDate = "2024-10-26";
        try {
            Field field = ScannerSubscription.class.getDeclaredField("m_maturityDateAbove");
            field.setAccessible(true);
            field.set(subscription, expectedDate);
        } catch (NoSuchFieldException e) {
            fail("Field m_maturityDateAbove not found.");
        }
        String actualDate = subscription.maturityDateAbove();
        assertEquals(expectedDate, actualDate);
    }

    @Test
    void testMaturityDateAbove_nullInput() throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {
        ScannerSubscription subscription = new ScannerSubscription();
        try {
            Field field = ScannerSubscription.class.getDeclaredField("m_maturityDateAbove");
            field.setAccessible(true);
            field.set(subscription, null);
        } catch (NoSuchFieldException e) {
            fail("Field m_maturityDateAbove not found.");
        }
        String actualDate = subscription.maturityDateAbove();
        assertNull(actualDate);
    }

    @Test
    void testMaturityDateAbove_emptyInput() throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {
        ScannerSubscription subscription = new ScannerSubscription();
        try {
            Field field = ScannerSubscription.class.getDeclaredField("m_maturityDateAbove");
            field.setAccessible(true);
            field.set(subscription, "");
        } catch (NoSuchFieldException e) {
            fail("Field m_maturityDateAbove not found.");
        }
        String actualDate = subscription.maturityDateAbove();
        assertEquals("", actualDate);
    }
}
