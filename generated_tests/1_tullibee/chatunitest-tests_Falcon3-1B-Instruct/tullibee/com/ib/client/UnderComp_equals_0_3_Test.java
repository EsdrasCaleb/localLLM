// Test method
package com.ib.client;

import org.junit.Test;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

public class UnderComp_equals_0_3_Test {

    @Test
    public void testEquals() {
        // Arrange
        UnderComp underComp1 = new UnderComp();
        UnderComp underComp2 = new UnderComp();
        // Act
        assertEquals(true, underComp1.equals(underComp2));
    }
}
