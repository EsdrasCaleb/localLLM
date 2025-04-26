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

class EWrapperMsgGenerator_accountDownloadEnd_12_0_Test {

    @Test
    void accountDownloadEnd_validAccountName_returnsCorrectMessage() {
        String accountName = "TestAccount";
        String expectedMessage = "accountDownloadEnd: TestAccount";
        String actualMessage = EWrapperMsgGenerator.accountDownloadEnd(accountName);
        assertEquals(expectedMessage, actualMessage);
    }

    @Test
    void accountDownloadEnd_nullAccountName_returnsCorrectMessage() {
        String accountName = null;
        String expectedMessage = "accountDownloadEnd: null";
        String actualMessage = EWrapperMsgGenerator.accountDownloadEnd(accountName);
        assertEquals(expectedMessage, actualMessage);
    }

    @Test
    void accountDownloadEnd_emptyAccountName_returnsCorrectMessage() {
        String accountName = "";
        String expectedMessage = "accountDownloadEnd: ";
        String actualMessage = EWrapperMsgGenerator.accountDownloadEnd(accountName);
        assertEquals(expectedMessage, actualMessage);
    }
}
