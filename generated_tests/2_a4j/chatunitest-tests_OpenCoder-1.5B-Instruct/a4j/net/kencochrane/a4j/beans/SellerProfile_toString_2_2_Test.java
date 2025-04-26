package net.kencochrane.a4j.beans;

import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class SellerProfile_toString_2_2_Test {

    @Test
    public void testToString() throws Exception {
        // Create a new instance of SellerProfile
        SellerProfile sellerProfile = new SellerProfile();
        // Call the toString method on the instance
        Method toStringMethod = SellerProfile.class.getDeclaredMethod("toString");
        if (Modifier.isPrivate(toStringMethod.getModifiers())) {
            toStringMethod.setAccessible(true);
        }
        String result = (String) toStringMethod.invoke(sellerProfile);
        // Expected output
        String expectedOutput = "SellerProfileDetails{firstName='John', lastName='Doe', ...}";
        // Assert that the result matches the expected output
        assertEquals(expectedOutput, result);
    }
}
