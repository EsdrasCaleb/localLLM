package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_moodyRatingAbove_10_0_Test {

    private ScannerSubscription mockScannerSubscription;

    @BeforeEach
    public void setup() {
        mockScannerSubscription = Mockito.mock(ScannerSubscription.class);
    }

    @Test
    public void testMoodyRatingAbove() {
        String testMoodyRating = "Test Moody Rating";
        Mockito.when(mockScannerSubscription.moodyRatingAbove()).thenReturn(testMoodyRating);
        assertEquals(testMoodyRating, mockScannerSubscription.moodyRatingAbove());
    }
}
