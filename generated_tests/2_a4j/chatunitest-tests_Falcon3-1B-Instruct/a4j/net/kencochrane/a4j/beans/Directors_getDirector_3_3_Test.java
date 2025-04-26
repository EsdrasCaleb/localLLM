package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Directors_getDirector_3_3_Test {

    // Focal class
    public class Directors {

        ArrayList<String> directors;

        // Method intention
        public String getDirector(int index) {
            String retString = null;
            if (directors.size() - 1 < index) {
                retString = (String) directors.get(index);
            }
            return retString;
        }

        // Method implementation
        public ArrayList<String> getDirectorsArray() {
            return directors;
        }
    }

    // Unit test for Directors method
    public class DirectorsTest {

        @Test
        public void testDirectorGetDirector() {
            Directors directors = new Directors();
            ArrayList<String> directorsArray = directors.getDirectorsArray();
            assertEquals("Director 1", directors.getDirector(0), "Directors should be ['Director 1', 'Director 2']");
            assertEquals("Director 3", directors.getDirector(2), "Directors should be ['Director 1', 'Director 2', 'Director 3']");
            assertEquals("Director 5", directors.getDirector(3), "Directors should be ['Director 1', 'Director 2', 'Director 3', 'Director 4', 'Director 5']");
            // Verify that Director 4 is not present
            assertNull(directors.getDirector(4));
        }
    }
}
