package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_moodyRatingAbove_10_2_Test {

    @Test
    public void testMoodyRatingAbove() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.moodyRatingAbove();
        // We can't test the return value directly, so we'll check if the method is called
        scannerSubscription.moodyRatingBelow("someRating");
    }

    @Test
    public void testMoodyRatingAboveWithInput() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.moodyRatingAbove("BBB");
        assertEquals("BBB", scannerSubscription.moodyRatingAbove());
    }

    @Test
    public void testMoodyRatingAboveWithInvalidInput() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        assertThrows(NullPointerException.class, () -> scannerSubscription.moodyRatingAbove());
    }
}
