package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_belowPrice_5_0_Test {

    @InjectMocks
    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testBelowPrice() throws Exception {
        // Set the private field m_belowPrice using reflection
        double expectedBelowPrice = 100.0;
        var field = ScannerSubscription.class.getDeclaredField("m_belowPrice");
        field.setAccessible(true);
        field.set(scannerSubscription, expectedBelowPrice);
        // Call the method under test
        double actualBelowPrice = scannerSubscription.belowPrice();
        // Assert the result
        assertEquals(expectedBelowPrice, actualBelowPrice, "The belowPrice method should return the value of m_belowPrice");
    }
}
