package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_maturityDateAbove_35_0_Test {

    @Test
    void testMaturityDateAbove_ValidDate() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.maturityDateAbove("2024-12-31");
        Field field = ScannerSubscription.class.getDeclaredField("m_maturityDateAbove");
        field.setAccessible(true);
        String actualDate = (String) field.get(subscription);
        assertEquals("2024-12-31", actualDate);
    }

    @Test
    void testMaturityDateAbove_NullDate() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.maturityDateAbove(null);
        Field field = ScannerSubscription.class.getDeclaredField("m_maturityDateAbove");
        field.setAccessible(true);
        String actualDate = (String) field.get(subscription);
        assertNull(actualDate);
    }

    @Test
    void testMaturityDateAbove_EmptyDate() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.maturityDateAbove("");
        Field field = ScannerSubscription.class.getDeclaredField("m_maturityDateAbove");
        field.setAccessible(true);
        String actualDate = (String) field.get(subscription);
        assertEquals("", actualDate);
    }

    @Test
    void testMaturityDateAbove_InvalidDate() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.maturityDateAbove("invalid date");
        Field field = ScannerSubscription.class.getDeclaredField("m_maturityDateAbove");
        field.setAccessible(true);
        String actualDate = (String) field.get(subscription);
        assertEquals("invalid date", actualDate);
    }
}
