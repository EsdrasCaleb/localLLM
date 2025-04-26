package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

public class EWrapperMsgGenerator_accountDownloadEnd_12_0_Test {

    @Test
    public void testAccountDownloadEnd_ValidAccountName() {
        // Arrange
        String accountName = "TestAccount";
        // Act
        String result = EWrapperMsgGenerator.accountDownloadEnd(accountName);
        // Assert
        assertEquals("accountDownloadEnd: TestAccount", result);
    }

    @Test
    public void testAccountDownloadEnd_EmptyAccountName() {
        // Arrange
        String accountName = "";
        // Act
        String result = EWrapperMsgGenerator.accountDownloadEnd(accountName);
        // Assert
        assertEquals("accountDownloadEnd: ", result);
    }

    @Test
    public void testAccountDownloadEnd_NullAccountName() {
        // Arrange
        String accountName = null;
        // Act
        String result = EWrapperMsgGenerator.accountDownloadEnd(accountName);
        // Assert
        assertEquals("accountDownloadEnd: null", result);
    }
}
