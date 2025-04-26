package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_spRatingAbove_33_0_Test {

    @Test
    void testSpRatingAbove_validInput() throws Exception {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.spRatingAbove("BBB+");
        Field field = ScannerSubscription.class.getDeclaredField("m_spRatingAbove");
        field.setAccessible(true);
        assertEquals("BBB+", field.get(subscription));
    }

    @Test
    void testSpRatingAbove_nullInput() throws Exception {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.spRatingAbove(null);
        Field field = ScannerSubscription.class.getDeclaredField("m_spRatingAbove");
        field.setAccessible(true);
        assertEquals(null, field.get(subscription));
    }

    @Test
    void testSpRatingAbove_emptyInput() throws Exception {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.spRatingAbove("");
        Field field = ScannerSubscription.class.getDeclaredField("m_spRatingAbove");
        field.setAccessible(true);
        assertEquals("", field.get(subscription));
    }

    @Test
    void testSpRatingAbove_differentInput() throws Exception {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.spRatingAbove("AA-");
        Field field = ScannerSubscription.class.getDeclaredField("m_spRatingAbove");
        field.setAccessible(true);
        assertEquals("AA-", field.get(subscription));
    }
}
