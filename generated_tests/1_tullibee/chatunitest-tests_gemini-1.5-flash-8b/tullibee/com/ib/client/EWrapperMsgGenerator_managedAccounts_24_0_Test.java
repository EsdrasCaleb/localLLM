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

class EWrapperMsgGenerator_managedAccounts_24_0_Test {

    @Test
    void managedAccounts_emptyAccounts() {
        String result = EWrapperMsgGenerator.managedAccounts("");
        assertEquals("Connected : The list of managed accounts are : []", result);
    }

    @Test
    void managedAccounts_singleAccount() {
        String result = EWrapperMsgGenerator.managedAccounts("account1");
        assertEquals("Connected : The list of managed accounts are : [account1]", result);
    }

    @Test
    void managedAccounts_multipleAccounts() {
        String result = EWrapperMsgGenerator.managedAccounts("account1,account2,account3");
        assertEquals("Connected : The list of managed accounts are : [account1,account2,account3]", result);
    }

    @Test
    void managedAccounts_nullInput() {
        String result = EWrapperMsgGenerator.managedAccounts(null);
        // Important:  Handling null input gracefully
        assertEquals("Connected : The list of managed accounts are : [null]", result);
    }
}
