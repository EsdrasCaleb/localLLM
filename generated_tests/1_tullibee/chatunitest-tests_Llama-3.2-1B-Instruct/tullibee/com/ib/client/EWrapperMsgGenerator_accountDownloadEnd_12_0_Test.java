package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_accountDownloadEnd_12_0_Test {

    @Mock
    private EWrapperMsgGenerator focal;

    @InjectMocks
    private EWrapperMsgGenerator instance;

    @Test
    public void testAccountDownloadEnd() {
        // Given
        String accountName = "TestAccount";
        // When
        String result = instance.accountDownloadEnd(accountName);
        // Then
        assertEquals("accountDownloadEnd: " + accountName, result);
    }
}
