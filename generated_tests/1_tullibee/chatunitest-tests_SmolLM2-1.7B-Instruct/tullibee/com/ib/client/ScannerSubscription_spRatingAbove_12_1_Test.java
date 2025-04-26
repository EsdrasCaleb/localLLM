package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_spRatingAbove_12_1_Test {

    @Test
    public void testSpRatingAbove() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.spRatingAbove("A");
        assertEquals("A", subscription.spRatingAbove());
    }

    @Test
    public void testSpRatingAbove_Null() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.spRatingAbove(null);
        assertNotNull(subscription.spRatingAbove());
    }
}
