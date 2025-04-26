package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_moodyRatingBelow_32_0_Test {

    @Test
    public void testMoodyRatingBelow() {
        ScannerSubscription ss = new ScannerSubscription();
        ss.moodyRatingBelow("BBB");
        assertEquals("BBB", ss.moodyRatingBelow());
    }

    @Test
    public void testMoodyRatingBelow_Null() {
        ScannerSubscription ss = new ScannerSubscription();
        ss.moodyRatingBelow(null);
        assertEquals(null, ss.moodyRatingBelow());
    }

    @Test
    public void testMoodyRatingBelow_EmptyString() {
        ScannerSubscription ss = new ScannerSubscription();
        ss.moodyRatingBelow("");
        assertEquals("", ss.moodyRatingBelow());
    }
}
