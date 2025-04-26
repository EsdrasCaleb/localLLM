package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_maturityDateBelow_15_0_Test {

    @InjectMocks
    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testMaturityDateBelow() {
        // Test when m_maturityDateBelow is null
        assertNull(scannerSubscription.maturityDateBelow());
        // Test when m_maturityDateBelow is set to a specific value
        String expectedMaturityDate = "2023-12-31";
        scannerSubscription.maturityDateBelow(expectedMaturityDate);
        assertEquals(expectedMaturityDate, scannerSubscription.maturityDateBelow());
    }
}
