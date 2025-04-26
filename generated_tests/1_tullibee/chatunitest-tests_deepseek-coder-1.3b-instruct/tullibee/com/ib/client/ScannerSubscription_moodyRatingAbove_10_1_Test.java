package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_moodyRatingAbove_10_1_Test {

    @Test
    public void testMoodyRatingAbove() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.moodyRatingAbove("A");
        assertEquals("A", subscription.moodyRatingAbove());
    }
}
