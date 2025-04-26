package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_moodyRatingBelow_11_1_Test {

    @Test
    public void testMoodyRatingBelow() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.moodyRatingBelow("BBB");
        assertEquals("BBB", scannerSubscription.moodyRatingBelow());
    }

    @Test
    public void testMoodyRatingBelow_Null() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.moodyRatingBelow("BBB");
        assertEquals("BBB", scannerSubscription.moodyRatingBelow());
    }

    @Test
    public void testMoodyRatingBelow_Empty() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.moodyRatingBelow("");
        assertEquals("", scannerSubscription.moodyRatingBelow());
    }

    @Test
    public void testMoodyRatingBelow_NullValue() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.moodyRatingBelow(null);
        assertEquals(null, scannerSubscription.moodyRatingBelow());
    }
}
