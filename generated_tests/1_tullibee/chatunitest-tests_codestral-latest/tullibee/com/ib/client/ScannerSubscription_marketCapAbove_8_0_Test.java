package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapAbove_8_0_Test {

    @InjectMocks
    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testMarketCapAbove() throws Exception {
        // Set up the private field m_marketCapAbove using reflection
        double expectedMarketCapAbove = 1000.0;
        scannerSubscription.marketCapAbove(expectedMarketCapAbove);
        // Invoke the focal method
        double actualMarketCapAbove = scannerSubscription.marketCapAbove();
        // Verify the result
        assertEquals(expectedMarketCapAbove, actualMarketCapAbove, 0.001);
    }
}
