package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_spRatingAbove_12_0_Test {

    @Test
    public void testSpRatingAbove() {
        ScannerSubscription scannerSubscription = Mockito.mock(ScannerSubscription.class);
        when(scannerSubscription.spRatingAbove()).thenReturn("AAA");
        String result = scannerSubscription.spRatingAbove();
        assertEquals("AAA", result);
    }
}
