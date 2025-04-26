package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_maturityDateAbove_14_0_Test {

    @Test
    public void testMaturityDateAbove() {
        // Create a mock object of ScannerSubscription
        ScannerSubscription scannerSubscription = Mockito.mock(ScannerSubscription.class);
        // Set the expected value for the "maturityDateAbove()" method
        String expectedMaturityDate = "2022-12-31";
        Mockito.when(scannerSubscription.maturityDateAbove()).thenReturn(expectedMaturityDate);
        // Call the "maturityDateAbove()" method and assert the result
        String actualMaturityDate = scannerSubscription.maturityDateAbove();
        assertEquals(expectedMaturityDate, actualMaturityDate);
    }
}
