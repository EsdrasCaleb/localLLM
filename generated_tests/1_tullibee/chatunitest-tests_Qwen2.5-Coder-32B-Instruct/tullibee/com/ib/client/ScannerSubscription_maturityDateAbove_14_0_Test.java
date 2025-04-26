package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_maturityDateAbove_14_0_Test {

    @InjectMocks
    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testMaturityDateAbove_DefaultValue() {
        // Given: Default value of m_maturityDateAbove is null
        String expected = null;
        // When: maturityDateAbove() is called
        String result = scannerSubscription.maturityDateAbove();
        // Then: It should return the default value
        assertEquals(expected, result);
    }

    @Test
    public void testMaturityDateAbove_SetValue() {
        // Given: Set a specific value to m_maturityDateAbove
        String expected = "20231231";
        scannerSubscription.maturityDateAbove(expected);
        // When: maturityDateAbove() is called
        String result = scannerSubscription.maturityDateAbove();
        // Then: It should return the set value
        assertEquals(expected, result);
    }
}
