package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_moodyRatingAbove_31_0_Test {

    @Test
    void testMoodyRatingAbove_validInput() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.moodyRatingAbove("Baa1");
        Field field = ScannerSubscription.class.getDeclaredField("m_moodyRatingAbove");
        field.setAccessible(true);
        assertEquals("Baa1", field.get(subscription));
    }

    @Test
    void testMoodyRatingAbove_nullInput() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.moodyRatingAbove(null);
        Field field = ScannerSubscription.class.getDeclaredField("m_moodyRatingAbove");
        field.setAccessible(true);
        assertNull(field.get(subscription));
    }

    @Test
    void testMoodyRatingAbove_emptyInput() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.moodyRatingAbove("");
        Field field = ScannerSubscription.class.getDeclaredField("m_moodyRatingAbove");
        field.setAccessible(true);
        assertEquals("", field.get(subscription));
    }

    @Test
    void testMoodyRatingAbove_differentInput() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.moodyRatingAbove("A1");
        Field field = ScannerSubscription.class.getDeclaredField("m_moodyRatingAbove");
        field.setAccessible(true);
        assertEquals("A1", field.get(subscription));
    }
}
