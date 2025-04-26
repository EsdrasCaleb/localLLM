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
    public void testAccountDownloadEnd() {
        // Test with a non-empty account name
        String accountName = "TestAccount";
        String expectedMessage = "accountDownloadEnd: TestAccount";
        String actualMessage = EWrapperMsgGenerator.accountDownloadEnd(accountName);
        assertEquals(expectedMessage, actualMessage, "The message for account download end should match the expected format.");
        // Test with an empty account name
        accountName = "";
        expectedMessage = "accountDownloadEnd: ";
        actualMessage = EWrapperMsgGenerator.accountDownloadEnd(accountName);
        assertEquals(expectedMessage, actualMessage, "The message for account download end should match the expected format even with an empty account name.");
        // Test with a null account name
        accountName = null;
        expectedMessage = "accountDownloadEnd: null";
        actualMessage = EWrapperMsgGenerator.accountDownloadEnd(accountName);
        assertEquals(expectedMessage, actualMessage, "The message for account download end should match the expected format even with a null account name.");
    }
}
