package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapBelow_9_2_Test {

    @Mock
    private ScannerSubscription mockScannerSubscription;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testMarketCapBelow() {
        double expectedMarketCapBelow = 1000000.0;
        when(mockScannerSubscription.marketCapBelow()).thenReturn(expectedMarketCapBelow);
        double actualMarketCapBelow = mockScannerSubscription.marketCapBelow();
        assertEquals(expectedMarketCapBelow, actualMarketCapBelow);
    }
}
