package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_maturityDateAbove_35_0_Test {

    @Test
    void testMaturityDateAbove() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        String maturityDate = "2024-10-26";
        subscription.maturityDateAbove(maturityDate);
        Field maturityDateField = ScannerSubscription.class.getDeclaredField("m_maturityDateAbove");
        maturityDateField.setAccessible(true);
        String actualMaturityDate = (String) maturityDateField.get(subscription);
        assertEquals(maturityDate, actualMaturityDate);
    }

    @Test
    void testMaturityDateAbove_NullInput() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        String maturityDate = null;
        subscription.maturityDateAbove(maturityDate);
        Field maturityDateField = ScannerSubscription.class.getDeclaredField("m_maturityDateAbove");
        maturityDateField.setAccessible(true);
        String actualMaturityDate = (String) maturityDateField.get(subscription);
        assertNull(actualMaturityDate);
    }

    @Test
    void testMaturityDateAbove_EmptyInput() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        String maturityDate = "";
        subscription.maturityDateAbove(maturityDate);
        Field maturityDateField = ScannerSubscription.class.getDeclaredField("m_maturityDateAbove");
        maturityDateField.setAccessible(true);
        String actualMaturityDate = (String) maturityDateField.get(subscription);
        assertEquals("", actualMaturityDate);
    }
}
