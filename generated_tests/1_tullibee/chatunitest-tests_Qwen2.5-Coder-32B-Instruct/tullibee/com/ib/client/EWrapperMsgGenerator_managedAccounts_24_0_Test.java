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
    @DisplayName("Test managedAccounts with non-empty account list")
    public void testManagedAccountsWithNonEmptyList() {
        // Arrange
        String accountsList = "ACC123,ACC456,ACC789";
        String expectedOutput = "Connected : The list of managed accounts are : [" + accountsList + "]";
        // Act
        String result = EWrapperMsgGenerator.managedAccounts(accountsList);
        // Assert
        assertEquals(expectedOutput, result);
    }

    @Test
    @DisplayName("Test managedAccounts with empty account list")
    public void testManagedAccountsWithEmptyList() {
        // Arrange
        String accountsList = "";
        String expectedOutput = "Connected : The list of managed accounts are : []";
        // Act
        String result = EWrapperMsgGenerator.managedAccounts(accountsList);
        // Assert
        assertEquals(expectedOutput, result);
    }

    @Test
    @DisplayName("Test managedAccounts with null account list")
    public void testManagedAccountsWithNullList() {
        // Arrange
        String accountsList = null;
        String expectedOutput = "Connected : The list of managed accounts are : [null]";
        // Act
        String result = EWrapperMsgGenerator.managedAccounts(accountsList);
        // Assert
        assertEquals(expectedOutput, result);
    }
}
