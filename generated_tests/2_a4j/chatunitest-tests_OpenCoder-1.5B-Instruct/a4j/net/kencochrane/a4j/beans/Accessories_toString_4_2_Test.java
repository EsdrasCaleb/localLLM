package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Accessories_toString_4_2_Test {

    @Test
    public void testToString() {
        // Create a mock object of the Accessories class
        Accessories accessories = Mockito.mock(Accessories.class);
        // Create a new ArrayList of Strings
        ArrayList<String> mockAccessoryList = new ArrayList<>();
        mockAccessoryList.add("Accessory 1");
        mockAccessoryList.add("Accessory 2");
        mockAccessoryList.add("Accessory 3");
        // Use Mockito's `when()` method to mock the `getAccessory()` method of the mock object
        when(accessories.getAccessory()).thenReturn(mockAccessoryList.toArray(new String[0]));
        // Call the `toString()` method of the mock object
        String result = accessories.toString();
        // Expected output should be a string with the details of the mock object
        String expectedOutput = "# of Accessories = 3\n" + "MiniProduct - Accessory 1\n" + "MiniProduct - Accessory 2\n" + "MiniProduct - Accessory 3\n";
        // Assert that the result matches the expected output
        assertEquals(expectedOutput, result);
    }
}
