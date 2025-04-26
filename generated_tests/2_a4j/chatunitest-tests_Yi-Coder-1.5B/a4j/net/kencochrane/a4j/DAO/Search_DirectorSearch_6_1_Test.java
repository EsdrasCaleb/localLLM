package net.kencochrane.a4j.DAO;

import java.lang.reflect.Method;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import com.wutka.jox.JOXBeanInputStream;
import net.kencochrane.a4j.beans.BlendedSearch;
import net.kencochrane.a4j.beans.ProductInfo;
import net.kencochrane.a4j.beans.SellerSearch;
import net.kencochrane.a4j.file.FileUtil;
import java.io.FileInputStream;

@DisplayName("Test class for the method DirectorSearch(String, String, String)")
class Search_DirectorSearch_6_1_Test {

    @Test
    void testDirectorSearch() {
        String directorName = "John";
        String mode = "lite";
        String page = "1";
        Search search = new Search();
        try {
            Method method = search.getClass().getDeclaredMethod("DirectorSearch", String.class, String.class, String.class);
            method.setAccessible(true);
            Object result = method.invoke(search, directorName, mode, page);
            System.out.println(result);
            assertNotNull(result);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
