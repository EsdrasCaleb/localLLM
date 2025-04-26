package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class Authors_getAuthor_3_0_Test {

    @Test
    public void testGetAuthorWithValidIndex() {
        // Create an instance of the Authors class
        Authors authors = new Authors();
        // Add some authors to the list
        authors.setAuthor(new String[] { "John Doe", "Jane Smith" });
        // Invoke the getAuthor method with a valid index
        String result = authors.getAuthor(0);
        // Verify the result
        assertEquals("John Doe", result);
    }

    @Test
    public void testGetAuthorWithInvalidIndex() {
        // Create an instance of the Authors class
        Authors authors = new Authors();
        // Add some authors to the list
        authors.setAuthor(new String[] { "John Doe", "Jane Smith" });
        // Invoke the getAuthor method with an invalid index
        String result = authors.getAuthor(-1);
        // Verify the result
        assertNull(result);
    }
}
