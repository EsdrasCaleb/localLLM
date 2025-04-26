package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Authors_toString_4_0_Test {

    @Test
    void toStringTest() {
        Authors authors = new Authors();
        String[] authorsArray = { "Author1", "Author2" };
        authors.setAuthor(authorsArray);
        String expectedOutput = "# of Authors = 2\n" + "Author - Author1\n" + "Author - Author2\n";
        assertEquals(expectedOutput, authors.toString());
    }
}
