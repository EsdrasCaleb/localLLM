package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class Authors_toString_4_0_Test {

    @Test
    public void testToString() {
        Authors authors = new Authors();
        authors.setAuthor(new String[] { "Alice", "Bob", "Charlie" });
        assertEquals("Authors is null or size 0\nAuthor - Alice\nAuthor - Bob\nAuthor - Charlie", authors.toString());
    }
}
