package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_spRatingBelow_13_0_Test {

    @Test
    void testSpRatingBelow() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.spRatingBelow("A");
        assertEquals("A", scannerSubscription.spRatingBelow());
    }
}
