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

public class EWrapperMsgGenerator_managedAccounts_24_0_Test {

    @Test
    public void testManagedAccounts() {
        String accountsList = "account1, account2, account3";
        String expectedMessage = "Connected : The list of managed accounts are : [" + accountsList + "]";
        String actualMessage = EWrapperMsgGenerator.managedAccounts(accountsList);
        assertEquals(expectedMessage, actualMessage);
    }
}
